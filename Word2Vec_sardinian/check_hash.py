import hashlib

def file_hash(filename):
    """ Calculate MD5 hash of a file. """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

filename = 'data/WikiText2/wikitext-2-v1.zip'
print("MD5:", file_hash(filename))