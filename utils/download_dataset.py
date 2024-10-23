import requests
import jsonlines
from tqdm import tqdm

def download_paperdoc(jsonline_file: str, cache_dir: str) -> None:
    """
    Download paperdoc dataset.
    """
    with jsonlines.open(jsonline_file) as reader:
        for obj in tqdm(reader):
            try:
                pdf_name = obj["PDF name"]
                url = f"https://arxiv.org/pdf/{pdf_name}"

                response = requests.get(url)

                with open(f"{cache_dir}/{pdf_name}.pdf", "wb") as f:
                    f.write(response.content)
                break
            except Exception as e:
                print(e)