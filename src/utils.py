import sys
sys.path.append("./contriever")
sys.path.append("./contriever/src")
from contriever.src.contriever import Contriever
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, AutoModel, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
import torch


model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "gtr-base": "sentence-transformers/gtr-t5-base",
    "gtr-base-v2t": "sentence-transformers/gtr-t5-base"

}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-ctx_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "gtr-base": "sentence-transformers/gtr-t5-base",
    "gtr-base-v2t": "sentence-transformers/gtr-t5-base"
}

def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def get_gtr_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def get_gtr_v2t_emb(model, input):
    model_output = model(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
    hidden_state = model_output.last_hidden_state
    embeddings = mean_pool(hidden_state, input['attention_mask'])
    return embeddings

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'dpr' in model_code:
        model = DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    elif model_code == 'gtr-base':
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        c_model = model
        tokenizer = model.tokenizer
        tokenizer.mask_token_id = tokenizer.pad_token_id
        get_emb = get_gtr_emb
    elif model_code == 'gtr-base-v2t':
        model = AutoModel.from_pretrained(model_code_to_qmodel_name[model_code]).encoder
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = get_gtr_v2t_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb