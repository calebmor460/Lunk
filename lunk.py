print("Starting LUNK(Large Universal Neural Kombiner), please wait...")
import torch, shutil, json, concurrent.futures, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import psutil, os
import gradio as gr
import numpy as np
import random
blend_ratio, fp16, always_output_fp16, max_shard_size, verbose_info, force_cpu, load_sharded = 0.5, False, True, "2000MiB", True, True, True
test_prompt, test_max_length = "Test, ", 32
blend_ratio_b = 1.0 - blend_ratio
def get_cpu_threads():
    try:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        return physical_cores, logical_cores
    except: return None
cpu_info = get_cpu_threads()
physical_cores, logical_cores = cpu_info if cpu_info else (4, 8)
def get_model_info(model):
    with torch.no_grad():
        outfo, cntent = "\n==============================\n", 0
        for name, para in model.named_parameters():
            cntent += 1
            outfo += ('{}: {}'.format(name, para.shape)) + "\n"
        outfo += ("Num Entries: " + str(cntent)) + "\n"
        outfo += ("==============================\n")
        return outfo
def merge_models(model1,model2,blend_ratio):
    with torch.no_grad():
        tensornum = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.shape == p2.shape:
                p1 *= (blend_ratio)
                p2 *= (blend_ratio_b)
                p1 += p2
                tensornum += 1
                print("Merging tensor "+str(tensornum))
            else: p1, p2 = p1, p2
def read_index_filenames(sourcedir):
    index = json.load(open(sourcedir + '/pytorch_model.bin.index.json', 'rt'))
    fl = [v for _, v in index['weight_map'].items()]
    return fl
def merge_models_and_save(model_path1, model_path2, model_path3=None):
    if not model_path1 or not model_path2:
        return "\nYou must select two directories containing models to merge and one output directory. Exiting."
    with torch.no_grad():
        if fp16:
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.float32)
        device = torch.device("cuda") if (torch.cuda.is_available() and not force_cpu) else torch.device("cpu")
        print("Loading Model 1...")
        model1 = AutoModelForCausalLM.from_pretrained(model_path1, torch_dtype='auto', low_cpu_mem_usage=True) #,torch_dtype=torch.float16
        model1 = model1.to(device)
        model1.eval()
        print("Model 1 Loaded. Dtype: " + str(model1.dtype))
        print("Loading Model 2...")
        model2 = AutoModelForCausalLM.from_pretrained(model_path2, torch_dtype='auto', low_cpu_mem_usage=True) #,torch_dtype=torch.float16
        model2 = model2.to(device)
        model2.eval()
        print("Model 2 Loaded. Dtype: " + str(model2.dtype))
        m1_info = get_model_info(model1)
        m2_info = get_model_info(model2)
        print("LUNKing models...")
        merge_models(model1, model2, blend_ratio)  # Pass the blend_ratio to merge_models function
        if model_path3:
            print("Saving new model...")
            newsavedpath = model_path3+"/converted_model"
            if always_output_fp16 and not fp16:
                model1.half()
            model1.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
            print("\nSaved to: " + newsavedpath)
        else:
            print("\nOutput model was not saved as no output path was selected.")
        print("\nScript Completed.")
current_directory = os.getcwd()
def interface(input_text1, input_text2,input_text3, blend_ratio_slider):  # Add the blend_ratio_slider parameter
    global blend_ratio
    blend_ratio = blend_ratio_slider  # Update the blend_ratio global variable
    merge_models_and_save(input_text1, input_text2, input_text3)
    return "Success! Models have been LUNKed."
iface = gr.Interface(
    fn=interface,
    inputs=[
        gr.inputs.Dropdown(choices=os.listdir(current_directory), label="FIRST model directory"),
        gr.inputs.Dropdown(choices=os.listdir(current_directory), label="SECOND model directory"),
        gr.inputs.Dropdown(choices=os.listdir(current_directory), label="OUTPUT model directory"),
        gr.inputs.Slider(minimum=0, maximum=1, step=0.01, default=0.5, label="Blend Ratio")
    ],
    outputs="text",
    title="LUNK(Large Universal Neural Kombiner)",
    description="Select some models and mash them into a new one! So long as they're the same size and architecture..",
)
iface.launch()
