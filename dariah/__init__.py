"""
This is a library for topic modeling and visualization.

Using the high-level API is easy:

```
>>> model, vis = dariah.topics(directory="corpus",
                               stopwords=100,
                               num_topics=10,
                               num_iterations=1000)
```

"""

from dariah.api import topics
