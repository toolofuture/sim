import requests
import json
import os

class RijksmuseumAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.rijksmuseum.nl/api/en/collection"

    def search_works(self, query="painting", type_filter="painting", involved_maker=None, ps=20, p=1):
        """
        Search for works in the collection.
        :param query: Search term
        :param p: Page number
        :param ps: Page size (max 100)
        """
        params = {
            "key": self.api_key,
            "q": query,
            "type": type_filter,
            "format": "json",
            "p": p,
            "ps": ps
        }
        if involved_maker:
            params["involvedMaker"] = involved_maker
            
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return None

    def get_work_details(self, object_number):
        """
        Get detailed information for a specific work.
        """
        url = f"{self.base_url}/{object_number}"
        params = {
            "key": self.api_key,
            "format": "json"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return None

    def download_image(self, image_url, save_path):
        """
        Download high-res image.
        """
        if not image_url:
            return False
        
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False

# 사용 예시 (Usage Example)
if __name__ == "__main__":
    # 실제 API 키를 입력해야 합니다. (Rijksstudio 환경설정에서 발급 가능)
    API_KEY = "YOUR_API_KEY_HERE" 
    
    api = RijksmuseumAPI(API_KEY)
    
    # 1. 'Rembrandt'의 작품 검색
    print("Searching for works by Rembrandt...")
    results = api.search_works(involved_maker="Rembrandt van Rijn", ps=5)
    
    if results and 'artObjects' in results:
        for obj in results['artObjects']:
            print(f"Title: {obj['title']}")
            print(f"Object Number: {obj['objectNumber']}")
            if obj['webImage']:
                print(f"Image URL: {obj['webImage']['url']}")
            print("-" * 30)
            
            # 상세 정보 및 고화질 이미지 URL 확인 (Tiles API 등을 위해 필요할 수 있음)
            # detail = api.get_work_details(obj['objectNumber'])
            # ...
    else:
        print("API 키가 유효하지 않거나 결과가 없습니다. 'YOUR_API_KEY_HERE'를 실제 키로 변경하세요.")
