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
    URL = "https://www.flipkart.com/hp-victus-amd-ryzen-5-hexa-core-6600h-8-gb-512-gb-ssd-windows-11-home-4-graphics-nvidia-geforce-rtx-3050-16-e1062ax-gaming-laptop/p/itm86ca62161fd3f?pid=COMGFA5FCNKUTPHG&lid=LSTCOMGFA5FCNKUTPHGMTIPBV&marketplace=FLIPKART&q=hp+victus&store=search.flipkart.com&srno=s_1_16&otracker=search&otracker1=search&fm=Search&iid=163e8c5c-9017-4abe-b581-24cf0649b52c.COMGFA5FCNKUTPHG.SEARCH&ppt=sp&ppn=sp&qH=8fa28d763aa5c003"
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
