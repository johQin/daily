# Selenium

# log

1. 通过selenium打开浏览器时间太长，并且报：`Exception managing chrome: error sending request for url (https://chromedriver.storage.googleapis.com/LATEST_RELEASE_103): operation timed out`，103是我的浏览器大版本

   - [参考](https://github.com/SeleniumHQ/selenium/issues/11406)

   - 参考链接告知安装selenium4.8.0

   - ```bash
      pip install selenium==4.8.0
     ```

   - 

2. [定位网页元素](https://blog.csdn.net/m0_54510474/article/details/121090473)

   ```python
   # find_element_by_css_selector() 被替换为find_element(“css selector”,"")
   driver.find_element("css selector",'#key')
   ```

   

3. 