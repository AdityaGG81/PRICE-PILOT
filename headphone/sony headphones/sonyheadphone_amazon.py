import pymysql
import requests
from bs4 import BeautifulSoup
import datetime

# ✅ Connect to MySQL Database
db = pymysql.connect(
    host="mysql.railway.internal",
    user="root",
    password="wtFMTvDDbZsRINtstBnbjvJyeARTNTdH",
    database="railway"
)
cursor = db.cursor()

# ✅ Function to Scrape and Store Amazon Data
def check_amazon_price():
    URL = 'https://www.amazon.in/Sony-WH-1000XM4-Cancelling-Headphones-Bluetooth/dp/B0863TXGM3/ref=sr_1_3?crid=NL085R246IF2&dib=eyJ2IjoiMSJ9.KiIeRLYdJ3HMV3b6j61tK3bfWoYKq1sYYs96aFpJRj6i87kuxnV3mQn6gx6YnGPIKKIl3B-VueD5zxbcjbAiY0p3RCMqUdw57EveWXWrL5JaP9OsZNnqXRrIBXuqAccOLXX1g1p_qG_Vd1SZGhglMMCApVxwCvkpC_jiWTAfKoEW0ABFaIS5U0tDBCHteHGc3g7R9RUAAJgMsDXYgyq2KGWAU-k5hvnbukLyXP4pSHA.jSicKMZXaAhLvpD5M5r29W1b-Uo89K1X9nmYYFY0Pf8&dib_tag=se&keywords=Sony%2BWH-1000XM4&nsdOptOutParam=true&qid=1743345152&sprefix=sony%2Bwh-1000xm4%2Caps%2C378&sr=8-3&th=1'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"}

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    product_name = soup.find(id="productTitle").get_text().strip()
    current_price = soup.find("span", class_="a-price-whole").get_text().replace(',', '').strip().rstrip('.')

    original_price_element = soup.find("span", class_="a-price a-text-price")
    original_price = original_price_element.find("span", class_="a-offscreen").get_text().replace('₹', '').replace(',', '').strip() if original_price_element else current_price

    date_scraped = datetime.date.today()

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
    values = (product_id, original_price, current_price, "Amazon", date_scraped)
    cursor.execute(query, values)
    db.commit()

    print(f"✅ Amazon Data Inserted:  ₹{current_price}")

check_amazon_price()

cursor.close()
db.close()
