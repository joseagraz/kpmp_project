from selenium import webdriver
from bs4 import BeautifulSoup
import time
import requests

# Set up Selenium with a headless browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

# Open the webpage with Selenium
driver.get('https://atlas.kpmp.org/repository/?size=n_20_n&filters%5B0%5D%5Bfield%5D=data_type&filters%5B0%5D%5Bvalues%5D%5B0%5D=Imaging&filters%5B0%5D%5Btype%5D=any&filters%5B1%5D%5Bfield%5D=redcap_id&filters%5B1%5D%5Bvalues%5D%5B0%5D=27-10039&filters%5B1%5D%5Btype%5D=any&sort%5B0%5D%5Bfield%5D=file_name_sort&sort%5B0%5D%5Bdirection%5D=asc')
driver.get('https://atlas.kpmp.org/api/v1/file/download/802f427a-dfe7-4347-ad2c-7148a9edcaf2/6a792e86-3b88-4e43-a0ab-70785420d5cf_MAX_21-0067_6%20AF_SHG_101822_Nav_Merge.tif')
# Wait for JavaScript to load
time.sleep(5)  # Adjust time as necessary

# Get the page source and close the browser
html = driver.page_source
driver.quit()

# Parse the HTML with BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
# soup.find_all('a', class_='download-link-class')
# Example: find all 'a' tags with a specific class (adjust as necessary)
for link in soup.find_all('a', class_='svg-inline--fa fa-download fas'):
    file_url = link.get('href')
    # Now you can use requests or any other method to download the file
    # Be sure to check the website's `robots.txt` and terms of service before downloading files
