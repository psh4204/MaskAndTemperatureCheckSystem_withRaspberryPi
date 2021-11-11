from selenium import webdriver
from bs4 import BeautifulSoup as soups
import os

def search_selenium(search_name, search_path, search_limit) :
    search_url = "https://www.google.com/search?q=" + str(search_name) + "&hl=ko&tbm=isch"
    
    browser = webdriver.Chrome('c:/chromedriver.exe')
    browser.get(search_url)
    
    image_count = len(browser.find_elements_by_tag_name("img"))
    
    print("로드된 이미지 개수 : ", image_count)

    browser.implicitly_wait(2)

    dir = os.path.join('dataCrwl')
    num = len(os.listdir(dir))
    for i in range( search_limit ) :
        image = browser.find_elements_by_tag_name("img")[i]
        image.screenshot("./dataCrwl/" + str(num+i) + ".png")
    browser.close()

if __name__ == "__main__" :

    search_name = input("검색하고 싶은 키워드(400장) : ")
    search_limit = 400
    search_path = "Your Path"
    # search_maybe(search_name, search_limit, search_path)
    search_selenium(search_name, search_path, search_limit)