from time import sleep
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from getpass import getpass
import datetime
from datetime import timedelta


from selenium.webdriver.common.by import By  # For Crawling
from selenium.webdriver.common.keys import Keys  # For Crawling
from selenium.webdriver.chrome.options import (
    Options,
)  # For setting some options for the driver, see Appendix.
from selenium.common.exceptions import NoSuchElementException  # Avoiding adds
from selenium.webdriver.support import expected_conditions as EC  # Conditions
from selenium.webdriver.support.ui import (
    WebDriverWait,
)  

def collect_tweet(card):
    try:
        date = card.find_element(By.XPATH, ".//time").get_attribute(
            "datetime"
        )  # Sponsored Content does not have this
    except NoSuchElementException:
        return None
    try:
        handle = card.find_element(By.XPATH, ".//a/div/div[1]/span/span").text
    except NoSuchElementException:
        return None    
    try:
        username = card.find_element(By.XPATH, ".//span[contains(text(),'@')]").text
    except NoSuchElementException:
        return None
    try:
        tweet_text = _collect_text(card)
    except NoSuchElementException:
        return None
    tweet = (handle, username, date, tweet_text)
    return tweet


def _collect_text(card):
    tweet_body = card.find_elements(By.XPATH, ".//div/div[2]/div[2]/div[2]/div/span[1]")
    text_list = [span.text for span in tweet_body]
    tweet_text = " "
    return tweet_text.join(text_list)


def my_scraper(max_tweets):
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    web_site = "https://twitter.com/home"
    driver.get(web_site)
    username = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@name='text']"))
    )
    username.send_keys(my_username)
    username.send_keys(Keys.RETURN)

    password = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
    )
    password.send_keys(my_password)
    password.send_keys(Keys.RETURN)

    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//input[@aria-label='Search query']")
        )
    )
    search_box.send_keys(
        f'"bitcoin" lang:en until:{d+timedelta(days=1)} since:{d} -filter:links -filter:replies'
    )
    search_box.send_keys(Keys.RETURN)
    # Scrape:

    data = []
    tweet_ids = set()  # In order to not collect duplicates
    last_position = driver.execute_script("return window.pageYOffset;")
    scrolling = True

    while scrolling:
        page_cards = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
        for card in page_cards[-95:]:
            tweet = collect_tweet(card)

            if tweet:
                tweet_id = "".join(tweet)

                if tweet_id not in tweet_ids:
                    tweet_ids.add(tweet_id)
                    data.append(tweet)

        # Loading bar VISUALIZATION
        percent_done = int((len(data) / max_tweets) * 100)
        print(f"{percent_done}% ", end="", flush=True)

        scroll_attempt = 0

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)
            curr_position = driver.execute_script("return window.pageYOffset;")

            if last_position == curr_position:
                scroll_attempt += 1

                # end of scroll region
                if scroll_attempt >= 3:
                    scrolling = False
                    break

                else:
                    sleep(2)  # attempt another scroll

            else:
                last_position = curr_position
                break

        if len(data) > max_tweets:
            scrolling = False

    # Close the web driver
    driver.close()
    return data