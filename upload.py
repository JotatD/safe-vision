from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/path/to/InternVL/internvl_chat/work_dirs/internvl_chat_v2_dpo/Internvl2-1B_1000",
    repo_id="hf-username/model-name",
    repo_type="model"
)
