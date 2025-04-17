# News Recommendation

## A starter backend program

### How to deploy

1. `git clone ...`
2. 

```bash
$ env\Scripts\activate  # Windows
$ . env/bin/activate  # Linux or macOS
```

3. 

```bash
$ python app.py
```

4. Open your browser and type  [http://localhost:5000](http://localhost:5000/) to access it.

5. Use `deactivate` to quit venv.

### Features

1. Displays a list of news articles
2. Allows user to 👍 Like / 👎 Dislike
3. Sends feedback to the backend
4. Builds and uses an inverted index
5. Backend prints out

### TODO

1. Find a way to gather articles
2. Sort articles by relevance (simple content-based recommender)
3. Save feedback to file or DB
4. Add a `/recommend` route to show top-N scored articles
