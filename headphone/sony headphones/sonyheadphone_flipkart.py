from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pymysql
import datetime

# ✅ Connect to MySQL Database
db = pymysql.connect(
    host="nozomi.proxy.rlwy.net",
        user="root",
        password="hkXIIviYSdzhcwvbDyLtyDBDkdnvhLGE",
        port=10649,
        database="railway"
)
cursor = db.cursor()

# ✅ Flipkart Scraper Function
def check_flipkart_price():
    URL = "https://www.flipkart.com/sony-wh-1000xm4-bluetooth/p/itm9f84f49ad6ac8?pid=ACCGCVGXQGFUNXTN&lid=LSTACCGCVGXQGFUNXTNMNI87P&marketplace=FLIPKART&q=Sony+WH-1000XM5&store=0pm%2Ffcn%2Fgc3%2Fka8&spotlightTagId=default_FkPickId_0pm%2Ffcn%2Fgc3%2Fka8&srno=s_1_1&otracker=search&otracker1=search&fm=Search&iid=6fb5980e-6727-4802-9da0-4bf94eacccae.ACCGCVGXQGFUNXTN.SEARCH&ppt=sp&ppn=sp&ssid=rey8bb6b5s0000001743345115858&qH=e1ff2b4872a0bf04"
    chromedriver_path = r"C:\Users\Parth\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service)

    # Open the webpage
    driver.get(URL)

    # ✅ Extract data
    product_name = driver.find_element(By.XPATH, '//span[@class="VU-ZEz"]').text.strip()
    current_price = driver.find_element(By.XPATH, '//div[contains(@class, "Nx9bqj")]').text.replace("₹", "").replace(",", "").strip()
    original_price = driver.find_element(By.XPATH, '//div[contains(@class, "yRaY8j")]').text.replace("₹", "").replace(",", "").strip()
    
    date_scraped = datetime.date.today()
    
    driver.quit()

    # ✅ Check if product exists
    cursor.execute("SELECT id FROM products WHERE product_name = %s", (product_name,))
    product = cursor.fetchone()

    if not product:
        cursor.execute("INSERT INTO products (product_name) VALUES (%s)", (product_name,))
        db.commit()
        product_id = cursor.lastrowid
    else:
        product_id = product[0]

    # ✅ Insert price data
    query = "INSERT INTO prices (product_id, original_price, current_price, source, date) VALUES (%s, %s, %s, %s, %s)"
    values = (product_id, original_price, current_price, "Flipkart", date_scraped)
    cursor.execute(query, values)
    db.commit()

    print(f"✅ Flipkart Data Inserted: {product_name}, ₹{current_price}, ₹{original_price}, {date_scraped}")

check_flipkart_price()

cursor.close()
db.close()
