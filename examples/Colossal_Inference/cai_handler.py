import json
import logging
import os
import zipfile
from abc import ABC

import torch
import transformers
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast
)

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import free_port  # assins a random port, for demo use only

from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
logger.info("ColossalAI version %s", colossalai.__version__)


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

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f"device set to {self.device}")


        # NOTE: zip model dir to model.zip
        #       => torch-model-archiver archives model.zip and extra files (e.g. config) to a .mar file 
        #       => torchserve
        #       => load and unpack .mar file to model.zip, config.json, etc
        #       => unzip model.zip and load model via from_pretrained
        model_dir = properties.get("model_dir")
        logger.info(f"Loading from model_dir {model_dir}")
        # Load inference config file
        filepath = model_dir + '/inference_config.json'
        with open(filepath, 'r') as file:
            config = json.load(file)
        # Load the model
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")
        logger.info(f"Loading {config['model_type']} pretrain model and tokenizer")
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
            logger.warning(f"Model type {config['model_type']} not supported yet.")

        logger.info("Transformer model from path %s loaded successfully", model_dir)
        logger.info(f"torch.cuda.device_count() {torch.cuda.device_count()}")

        self.model.half()
        self.model.cuda()
        self.model.eval()

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        rank = int(os.getenv("RANK", 0))
        host = os.getenv("MASTER_ADDR", 'localhost')
        port = os.getenv("MASTER_PORT", free_port())  # use a random free port
        logger.info(
            f"  local_rank {local_rank}"
            f"  world_size {world_size}"
            f"  rank {rank}"
            f"  host {host}"
            f"  port {port}"
        )

        torch.cuda.set_device(local_rank)

        colossalai.launch(config={}, rank=rank, world_size=world_size, host=host, port=port, backend='nccl')
        
        logger.info("Initializing TPInferEngine ...")
        shard_config = ShardConfig(enable_tensor_parallelism=True if config["tp_size"] > 1 else False, inference_only=True)
        self.infer_engine = TPInferEngine(self.model, shard_config, config['max_batch_size'], config['max_input_len'], config['max_output_len'])
        logger.info("TPInferEngine initialized successfully")

        self.model = self.infer_engine.model
        self.initialized = True


    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        logger.info("Pre-processing requests")
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

        logger.info(f"Generated text: {inferences}", )

        return inferences


    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
