import time
import scrapy
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import numpy as np

            
class HermersSpider(scrapy.Spider):
    name = 'hermers'
    start_urls = [
        # 'https://www.hermes.com/us/en/category/women/bags-and-small-leather-goods/bags-and-clutches/#||Category',
        'https://www.hermes.com/hk/en/category/women/bags-and-small-leather-goods/bags-and-clutches/#||Category',
        'https://www.hermes.cn/cn/zh/category/%E5%A5%B3%E5%A3%AB/%E7%AE%B1%E5%8C%85%E5%B0%8F%E7%9A%AE%E5%85%B7/%E7%AE%B1%E5%8C%85%E6%99%9A%E5%AE%B4%E5%8C%85/#||%E4%BA%A7%E5%93%81%E7%B3%BB%E5%88%97'
        ]
    
    def __init__(self) -> None:
        super().__init__()
        self.loca = {
            'cn': {
                'target': [('bolide', '迷你'), ('lindy', '迷你'), ('bolide', 'mini')],
                'item_url': 'https://www.hermes.cn/cn/zh/product/{}-{}',  # name, id
                'image_src': 'https://assets.hermes.cn/is/image/hermesproduct/{}_front_1?a=a&size=3000%2C3000&extend=300%2C300%2C300%2C300&align=0%2C0&$product_item_grid_b$=&resMode=&wid={}&hei={}'
            },
            'hk': {
                'target': [('picotin', '18'), ('picotin', '22')],
                'item_url': 'https://www.hermes.com/hk/en/product/{}-{}',  # name, id
                'image_src': 'https://assets.hermes.com/is/image/hermesproduct/{}_front_1?a=a&size=3000%2C3000&extend=300%2C300%2C300%2C300&align=0%2C0&$product_item_grid_b$=&wid={}&hei={}'
            }
        }
        self.to_addrs = {
            'cn': ['3328732477@qq.com', 'zoulijia124@163.com'],
            'hk': ['3328732477@qq.com', 'zoeedeng@163.com',]
        }

    def parse(self, response):
        bags = response.css(".product-item-name::text").extract()
        # bags = [b.lower() for b in bags]
        url = response.url
        location = url.split('/')[3]
        print(f'Url: {url}')
        print(f'Bags: {np.array(list(set(bags))).reshape(-1, 1)}')
        target = self.loca[location]['target']
        for (t, s) in target:
            stop = False
            for b in bags:
                if t.lower() in b.lower() and s.lower() in b.lower():
                    image_ids = response.xpath(f'//img[@alt="{b}"]/@id').extract()
                    widths = response.xpath(f'//img[@alt="{b}"]/@width').extract()
                    heights = response.xpath(f'//img[@alt="{b}"]/@height').extract()
                    
                    srcs = []
                    bag_buy_urls = []
                    for image_id, width, height in zip(image_ids, widths, heights):
                        if location == 'cn':
                            src_im_id = image_id[5:-3]
                        elif location == 'hk':
                            src_im_id = image_id[5:]
                        src = self.loca[location]['image_src'].format(src_im_id, width, height)
                        # src = self.image_src.format(image_id[5:], width, height)
                        srcs.append(src)
                        buy_name = b.replace(' ', '-')
                        buy = self.loca[location]['item_url'].format(buy_name, image_id[4:])
                        bag_buy_urls.append(buy)
                    # print(srcs)
                    # print(bag_buy_urls)
                    
                    # send email
                    smtp_obj = smtplib.SMTP_SSL()
                    smtp_obj.connect(
                        host="smtp.qq.com",
                        port=465,
                    )
                    msg = f"""
                        <h2 align='center'>We got {b}!</h2>\n"""
                    for src, buy, w, h in zip(srcs, bag_buy_urls, widths, heights):
                        w = str(min(int(w), 300))
                        h = str(min(int(h), 300))
                        img_html = f"""
                        <p align='center'><a href="{buy}"><b>Buy this one!</b></a></p>
                        <p align='center'>
                            <img src={src} alt={b} width='{w}' height='{h}'>
                        </p>
                        <br/>
                            """
                        msg += img_html
                    
                    msg += f"""<p><a href='{url}'>Check all available bags!</a></p>"""
                    # print(msg)
                    # import pdb; pdb.set_trace()
                    
                    message = MIMEText(msg, 'html', 'utf-8')
                    message['From'] = Header("Spider-Man", 'utf-8')   # 发送者
                    message['To'] = Header("Spider-Man", 'utf-8')        # 接收者
                    
                    subject = f'Friendly Spider Man: check "{b}"" now!'
                    message['Subject'] = Header(subject, 'utf-8')

                    res = smtp_obj.login(
                        user="3328732477@qq.com",
                        password="dklkhomwuezjcjfe"
                    )
                    
                    to_addrs = self.to_addrs[location]
                    smtp_obj.sendmail(
                        from_addr='3328732477@qq.com',
                        to_addrs=to_addrs,
                        msg=message.as_string())
                    stop = True
                    break
            if stop:
                break