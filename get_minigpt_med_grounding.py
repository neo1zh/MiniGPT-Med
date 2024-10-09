import argparse
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import os

import warnings
warnings.filterwarnings("ignore")
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="/home/yiyangai/Projects/zihao_zhao/LLaVA/ITGrad/MiniGPT_med/eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def get_grounding(img_path, prompt='Please describe the image.'):

    # ========================================
    #             Model Initialization
    # ========================================

    # print('Initializing model')
    args = parse_args()
    cfg = Config(args)
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = 'cuda:{}'.format(args.gpu_id)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    # print('Model Initialization Finished')

    # upload image
    chat_state = CONV_VISION_minigptv2.copy()
    img_list = []
    llm_message = chat.upload_img(img_path, chat_state, img_list)
    # print(llm_message)

    # ask a question
    user_message = "[grounding]" + prompt
    chat.encode_img(img_list)
    chat.ask(user_message, chat_state)

    # get answer
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=args.num_beams,
                              temperature=args.temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    # print(llm_message)
    return llm_message

if __name__ == '__main__':
    img_path = '/home/yiyangai/Projects/zihao_zhao/Data/eval/iu_xray/images/CXR1_1_IM-0001/0.png'
    print(get_grounding(img_path))
    