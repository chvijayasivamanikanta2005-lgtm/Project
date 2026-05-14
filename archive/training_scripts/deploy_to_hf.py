import os
try:
    from huggingface_hub import HfApi
except ImportError:
    print("❌ huggingface_hub is not installed. Please install it first:")
    print("   pip install huggingface_hub")
    exit(1)

def deploy():
    print("🚀 EV Dashboard - Hugging Face Deployment Script")
    print("-------------------------------------------------")
    
    repo_id = input("Enter your Hugging Face Space Repo ID (e.g., username/ev-dashboard): ").strip()
    if not repo_id:
        print("❌ Repo ID is required.")
        return
        
    api = HfApi()
    
    # Check if repo exists or create it
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"✅ Found existing Space: {repo_id}")
    except Exception:
        print(f"⚠️ Space '{repo_id}' not found.")
        ans = input(f"Do you want to create a new Streamlit Space named '{repo_id}'? (y/n): ")
        if ans.lower().startswith('y'):
            try:
                api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit")
                print(f"✅ Created new Streamlit Space: {repo_id}")
            except Exception as e:
                print(f"❌ Failed to create space. Are you logged in? Run 'huggingface-cli login'. Error: {e}")
                return
        else:
            print("Deployment cancelled.")
            return

    print("📤 Uploading repository files... This might take a minute depending on your connection.")
    
    # Files/folders to ignore during upload
    ignore_patterns = [
        "*.pyc", "__pycache__/", ".git/", ".env", 
        ".python-version", "deploy_to_hf.py", ".DS_Store",
        ".gemini/", "venv/"
    ]
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Deploy Streamlit EV BMS Dashboard"
        )
        print(f"🎉 Deployment successful!")
        print(f"🌍 View your app live at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed. Make sure you have write access. Error: {e}")
        print("Tip: Run 'huggingface-cli login' to authenticate if you haven't already.")

if __name__ == "__main__":
    deploy()
