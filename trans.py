from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

def translate():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-infobars")
    options.add_argument("start-maximized")
    options.add_argument("--disable-extensions")
    # Pass the argument 1 to allow and 2 to block
    options.add_experimental_option("prefs", { 
        "profile.default_content_setting_values.notifications": 1 ,
        "profile.managed_default_content_settings.images": 2
    })
    chrome_driver_binary = "/home/thanhdat/chromedriver"
    browser = webdriver.Chrome(chrome_driver_binary, options=options)

    words = open('words.txt').read().split('\n')
    words = [x.strip() for x in words]

    word2class = {}
    for word in words:
        try:
            browser.get(f"https://www.merriam-webster.com/dictionary/{word}")
            elms = browser.find_elements_by_class_name("fl")
            for elm in elms:
                text = elm.get_attribute("innerHTML")
                if text.contains("")

            print(f"{word} {word_type}")
        except:
            print("error word ", word)
    
    browser.close()

if __name__ == '__main__':
    translate()
