from bs4 import BeautifulSoup
import scrapy

class LNNodeParser(scrapy.Spider):
    name = "ln_node_parser"
    
    def __init__(self, nodes, **kwargs):
        self.nodes = nodes
        super().__init__(**kwargs)
    
    def start_requests(self):
        urls = ["https://1ml.com/node/%s" % n for n in self.nodes]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = '%s.html' % page
        with open("/mnt/idms/fberes/data/bitcoin_ln_research/1ml/%s" % filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

def get_info_type(li_item):
    info_type = "other"
    for t in li_item.span.attrs["class"]:
        if "icon-" in t:
            info_type = t.replace("icon-","")
            break
    return (info_type, li_item.get_text())
    
def get_node_info(info_part):
    if info_part == None:
        return dict([])
    else:
        return dict(get_info_type(item) for item in info_part.find_all("li"))

def extract_labels(labels_part):
    if labels_part == None:
        return None
    else:
        labels = []
        for item in labels_part.find_all("a"):
            labels.append(item.get_text())
        return labels
    
def extract_node_meta_data(node_id):
    with open("/mnt/idms/fberes/data/bitcoin_ln_research/1ml/%s.html" % node_id) as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    info_part = soup.find('ul', {"class":"wordwrap"})
    labels_part = soup.find('ul', {"class":"tags"})
    title_part = soup.find('title').get_text()
    meta_data = get_node_info(info_part)
    meta_data["labels"] = extract_labels(labels_part)
    meta_data["alias"] = title_part.split(" ")[1]
    meta_data["pub_key"] = node_id
    return meta_data