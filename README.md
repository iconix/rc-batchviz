setup bash on windows:
```bash
$ python -m venv batchviz
$ source batchviz/Scripts/activate
$ pip install -r requirements.txt
```

update `requirements.txt` after adding a new package:
```bash
$ pip freeze > requirements.txt
```

extra downloads for some `TopicModeler` methods:
```bash
$ python -m textblob.download_corpora
$ python -m nltk.downloader punkt_tab
```

common ways to run script:
```bash
# `w1+24_introductions.csv` sourced from a private file
$ time python main.py w1+24_introductions.csv --topic-method tfidf --components 3 --pseudonymize --preserve-names 'Nadja Rhodes'
```
