class Rating:
    def __init__(self, user_id, movie_id, rating):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating

    def __str__(self):
        return f"Rating(user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating})"

    def __repr__(self):
        return self.__str__()

    def __tuple__(self):
        return (self.user_id, self.movie_id), self.rating