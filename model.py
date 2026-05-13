import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


class PatentSimilarityModel(nn.Module):

    def __init__(self, model_name, token, use_bf16): # change default token

        super().__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if use_bf16
                else torch.float16
            ),

            bnb_4bit_use_double_quant=True,
        )

        self.backbone = AutoModel.from_pretrained(
            model_name,

            quantization_config=bnb_config,

            device_map="auto",

            output_hidden_states=True,
            
            token=token,

            dtype=(
                torch.bfloat16
                if use_bf16
                else torch.float16
            )
        )

        hidden_size = self.backbone.config.hidden_size

        # last-4-layer pooling regression head
        self.regressor = nn.Sequential(

            nn.Linear(hidden_size, hidden_size // 2),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(hidden_size // 2, 1)
        )

        self.backbone.config.output_hidden_states = True
        self.backbone.config.return_dict = True

        self.backbone = prepare_model_for_kbit_training(
            self.backbone
        )
        self.backbone.config.use_cache = False 

        lora_config = LoraConfig(

            r=16,

            lora_alpha=32,

            target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            ],

            lora_dropout=0.05,

            bias="none",

            task_type="FEATURE_EXTRACTION",
        )

        self.backbone = get_peft_model(
            self.backbone,
            lora_config
        )

        self.backbone.print_trainable_parameters()

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise ValueError("hidden_states is None. Check model config/output flags.")

        stacked = torch.stack([
            hidden_states[-1],
            hidden_states[-2],
            hidden_states[-3],
            hidden_states[-4],
        ]).mean(0)

        seq_lengths = attention_mask.sum(dim=1) - 1

        pooled = stacked[
            torch.arange(stacked.size(0)),
            seq_lengths
        ]

        logits = self.regressor(pooled)

        logits = logits.squeeze(-1)

        return logits
