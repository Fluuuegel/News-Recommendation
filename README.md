# News Recommendation

KTH DD2477 Project.

### How to deploy


1. `git clone ...`

2. Create a virtual environment: `python -m venv env`

3. 
    ```bash
    $ env\Scripts\activate  # Windows
    $ . env/bin/activate  # Linux or MacOS
    ```

4. `pip install -r requirements.txt`

5. Start Docker.

6. Run:

   ```bash
   $ docker run -d ` # Windows
     --name elasticsearch `
     -p 9200:9200 `
     -e "discovery.type=single-node" `
     -e "xpack.security.enabled=false" `
     docker.elastic.co/elasticsearch/elasticsearch:8.12.2
   ```

   ```bash
   $ docker run -d \ # Linux or MacOS
     --name elasticsearch \
     -p 9200:9200 \
     -e "discovery.type=single-node" \
     -e "xpack.security.enabled=false" \
     docker.elastic.co/elasticsearch/elasticsearch:8.12.2
   ```

7. Run:

    ```bash
    $ python build_tf_idf.py   // this is to get the tf-idfs before use
    $ python indexer.py
    ```

    Wait for minutes.

8. Open a new terminal and run:

    ```bash
    $ python app.py
    ```

9. Open your browser and type  [http://localhost:5000](http://localhost:5000/) to access it.

10. Use `deactivate` to quit venv.

### Features

1. Using Elasticsearch to crawl the New York Times for different categories of news.
2. Search and display results.
3. Allows user to 👍 Like / 👎 Dislike.
4. Recommend different news to users based on their preferences.

### Notes

1. The original `indexer.py` is a mess so I re-arrange the structure with the assistance of ChatGPT.
