# News Recommendation

## A starter backend program

### How to deploy


# 
pip install newspaper3k
pip install scikit-learn



1. `git clone ...`

2. 

    ```bash
    $ env\Scripts\activate  # Windows
    $ . env/bin/activate  # Linux or macOS
    ```

3. Start Docker

4. Run:

   ```bash
   $ docker run -d \
     --name elasticsearch \
     -p 9200:9200 \
     -e "discovery.type=single-node" \
     -e "xpack.security.enabled=false" \
     docker.elastic.co/elasticsearch/elasticsearch:8.12.2
   ```

5. Run:

    ```bash
    python indexer.py
    ```

    Wait for retriving

6. Open a new terminal and run:

    ```bash
    $ python app.py
    ```

7. Open your browser and type  [http://localhost:5000](http://localhost:5000/) to access it.

8. Use `deactivate` to quit venv.

### Features

1. Using Elasticsearch to crawl the New York Times for different categories of news
2. Search and display results
3. Allows user to 👍 Like / 👎 Dislike
4. Recommend different news to users based on their preferences

### TODO
