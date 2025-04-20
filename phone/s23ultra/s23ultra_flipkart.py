from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pymysql
import datetime

# ✅ Connect to MySQL Database
db = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="scraper_db"
)
cursor = db.cursor()

# ✅ Flipkart Scraper Function
def check_flipkart_price():
    URL = "https://www.flipkart.com/samsung-galaxy-s23-ultra-5g-phantom-black-256-gb/p/itm15952643ba06d?pid=MOBGMFFX6E6SGYWC&lid=LSTMOBGMFFX6E6SGYWCHKRFZT&marketplace=FLIPKART&q=samsung+s23+ultra+5g&store=tyy%2F4io&srno=s_1_2&otracker=AS_QueryStore_OrganicAutoSuggest_2_12_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_2_12_na_na_na&fm=Search&iid=59786240-6508-4f00-b3a9-679d4d10c2e4.MOBGMFFX6E6SGYWC.SEARCH&ppt=sp&ppn=sp&ssid=z8mw5euxrk0000001743337491350&qH=8c1eb023d39726ef"
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
