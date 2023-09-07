import requests
def main():
    blob_url = 'https://cdn.stability.ai/assets/org-rp8fWa6VaxuEKJ2Gw27Qm9Hh/00000000-0000-0000-0000-000000000000/5b799b5b-0eb9-48c4-e6da-4606352cc872'

    # Fetch the image data from the blob URL
    response = requests.get(blob_url)

    if response.status_code == 200:
        print(response.content)

main()