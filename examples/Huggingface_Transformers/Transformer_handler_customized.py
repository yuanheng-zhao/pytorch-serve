import ast
import json
import logging
import os
import zipfile
from abc import ABC

import torch
import transformers
from captum.attr import LayerIntegratedGradients
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GPT2TokenizerFast,
    LlamaForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast
)

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.shardformer import ShardConfig

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersTestHandler(BaseHandler, ABC):
    """
    Transformers handler class for testing
    """

    def __init__(self):
        super(TransformersTestHandler, self).__init__()
        self.initialized = False
        self.infer_engine = None


    def initialize(self, ctx):
        """Expected behaviour: the sharded Bloom/Llama model is loaded.

        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        logger.info(f"Loading from model_dir {model_dir}")
        print(f"Loading from model_dir {model_dir}")
        filepath = model_dir + '/inference_config.json'
        with open(filepath, 'r') as file:
            config = json.load(file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        # NOTE: .bin/.json/config => zip => MAR => zip => unzip => load model by from_pretrained
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        if config["model_type"] == "bloom":
            print("loading bloom pretrain model and tokenizer")
            self.model = BloomForCausalLM.from_pretrained(
                model_dir + "/model",
            )
            self.tokenizer = BloomTokenizerFast.from_pretrained(
                model_dir + "/model", return_tensors="pt"
            )
        elif config["model_type"] == "llama":
            print("loading llama pretrain model and tokenizer")
            self.model = LlamaForCausalLM.from_pretrained(
                model_dir + "/model",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir + "/model", return_tensors="pt"
            )
        else:
            print(f"config['model_type'] {config['model_type']} not supported yet.")

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.model = self.model.half()
        self.model.cuda()
        self.model.eval()

        logger.info("Initializing TPInferEngine ...")

        # FIXME might error out when launching colossalai
        # colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
        colossalai.launch_from_torch(config={})
        
        shard_config = ShardConfig(enable_tensor_parallelism=True if config["tp_size"] > 1 else False, inference_only=True)
        self.infer_engine = TPInferEngine(self.model, shard_config, config['max_batch_size'], config['max_input_len'], config['max_output_len'])
        # self.model = self.infer_engine.model

        logger.info("TPInferEngine initialized successfully")

        self.initialized = True


    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        logger.info("Pre-processing requests", requests)
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            logger.info("Received text: '%s'", input_text)

            inputs = self.tokenizer.encode_plus(
                input_text,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)


    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []

        # mode: text_generation
        input_ids_batch = input_ids_batch.to(self.device)
        outputs = self.infer_engine.generate(
            input_ids_batch,
            do_sample=False,
            # top_p=0.95,
            # top_k=60,
        )

        logger.info(f"Generated outputs: {outputs}")

        for i, _ in enumerate(outputs):
            inferences.append(
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            )

        logger.info("Generated text: '%s'", inferences)
        print("Generated text", inferences)

        return inferences


    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
