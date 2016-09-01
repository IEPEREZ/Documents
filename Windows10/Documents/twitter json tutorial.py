import urllib2
import simplejson

url = "https://twitter.com/twitterapi/status/745596132462571520?lang=en"

if __name__ == "__main__":
    req = urllib2.Request(url)
    opener = urllib2.build_opener()
    f = opener.open(req)
    json = simplejson.load(f)
    
    for item in json:
        print item.get('created_at')
        print item.get('text')
        