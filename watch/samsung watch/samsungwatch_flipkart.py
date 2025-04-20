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
    URL = "https://www.flipkart.com/samsung-galaxy-watch6-classic-lte/p/itm0188047c07b62?pid=SMWGRV9YWTNNAVA7&lid=LSTSMWGRV9YWTNNAVA7ZYGEUV&marketplace=FLIPKART&q=amsung+Galaxy+Watch6+Classic+LTE&store=ajy%2Fbuh&srno=s_1_1&otracker=search&otracker1=search&fm=Search&iid=4eb011c6-538f-41a5-b17e-f654ba4ca26e.SMWGRV9YWTNNAVA7.SEARCH&ppt=sp&ppn=sp&ssid=9dbxjg8wpc0000001743344701526&qH=93d4df6cd835aeef"
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
