import pymysql
import requests
from bs4 import BeautifulSoup
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

# ✅ Function to Scrape and Store Amazon Data
def check_amazon_price():
    URL = 'https://www.amazon.in/Samsung-Galaxy-Ultra-Phantom-Storage/dp/B0BT9FDZ8N/ref=sr_1_8?crid=2RZS7GLOOJGE1&dib=eyJ2IjoiMSJ9.H7DO_zWJTPmVY_saHBeX4Tpckgywng2xzObKdkkpgC2d4yR8Ha7bBy-R1nxP7PNByWTReeSb4gVAJDVb03MoSgZADGK8w1htAqapS_3sowDcdXhg2E46wAhEGBHRlKSklV4pkZhWFJR-g39bzf9rH-u0LOquJeGJD519fTkj3UD5kNh0K9YAsdzUlhiIR1FhZnBaIJmhH75AHRc4nvELS44fy8wSkxUsdsOvR6L4-fU.BnhCL2IxtDE2_o8WsqQVXi37f3S1nC-bn9UUnt4kfGI&dib_tag=se&keywords=samsung%2Bs23&nsdOptOutParam=true&qid=1743337419&sprefix=samsung%2Bs23%2B%2Caps%2C360&sr=8-8&th=1'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"}

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    product_name = soup.find(id="productTitle").get_text().strip()
    current_price = soup.find("span", class_="a-price-whole").get_text().replace(',', '').strip()

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

    print(f"✅ Amazon Data Inserted: {product_name}, ₹{current_price}, ₹{original_price}, {date_scraped}")

check_amazon_price()

cursor.close()
db.close()
