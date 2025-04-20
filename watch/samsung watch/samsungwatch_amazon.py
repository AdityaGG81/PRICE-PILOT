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
    URL = 'https://www.amazon.in/Samsung-Galaxy-Classic-Compatible-Android/dp/B0CCV98L46/ref=sr_1_7?crid=2HZ0S9W7WN7KL&dib=eyJ2IjoiMSJ9.Wre-HYsu8SMJzaoHpyIWzN-l1POWgFmXU7vmoAUp7d2VeWFsCt1jQkTT08FQUFCMvUgb7uOZ6knGL9ETgmgDEj-hNnKGOshUwqT8O4E-djK5KVIh5iiMPF9AXcLN16ipnxlaIplknP9dump5ExLZns3xPGkxJa4t_SwXkDVuBwQkj-fVUVI8v7tqShRfwqW-yyIFej4OXkzTbGdNiahdz7gAc0fxm26Bs0p6FFX9pCc.COTU8Z80dLgDUNLjJTPOfl2N-NZEcLpAW4mY7R40Occ&dib_tag=se&keywords=Samsung%2BGalaxy%2BWatch%2B6&nsdOptOutParam=true&qid=1743344568&sprefix=samsung%2Bgalaxy%2Bwatch%2B6%2Caps%2C389&sr=8-7&th=1'
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
