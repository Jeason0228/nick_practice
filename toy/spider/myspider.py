import time
import scrapy
from scrapy.mail import MailSender
import smtplib
from email.mime.text import MIMEText
from email.header import Header

            
class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://www.hermes.com/us/en/category/women/bags-and-small-leather-goods/bags-and-clutches/#||Category']
    
    def __init__(self) -> None:
        super().__init__()
        self.url = 'https://www.hermes.com/us/en/category/women/bags-and-small-leather-goods/bags-and-clutches/#||Category'
        self.passW = "dklkhomwuezjcjfe"
        self.target = ['picotin', 'Dog']
        self.last_email = time.time()

    def parse(self, response):
        # bags = response.xpath("//h3/text()").extract()
        bags = response.css(".product-item-name::text").extract()
        # import pdb; pdb.set_trace()
        for t in self.target:
            stop = False
            for b in bags:
                if t.lower() in b.lower():
                    # send email
                    smtp_obj = smtplib.SMTP_SSL()
                    smtp_obj.connect(
                        host="smtp.qq.com",
                        port=465,
                    )
                    
                    msg = """
                        <p>菜篮子有了...</p>
                        <p><a href="https://www.hermes.com/us/en/category/women/bags-and-small-leather-goods/bags-and-clutches/#||Category">Buy!</a></p>
                        """
                    message = MIMEText(msg, 'plain', 'utf-8')
                    message['From'] = Header("吴彦祖", 'utf-8')   # 发送者
                    message['To'] =  Header("吴彦祖", 'utf-8')        # 接收者
                    
                    subject = '!菜篮子有啦'
                    message['Subject'] = Header(subject, 'utf-8')

                    res = smtp_obj.login(
                        user="3328732477@qq.com",
                        password="dklkhomwuezjcjfe"
                    )
                    
                    smtp_obj.sendmail(from_addr='3328732477@qq.com', to_addrs='3328732477@qq.com',
                                    msg=message.as_string())
                    stop = True
                    break
            if stop:
                break