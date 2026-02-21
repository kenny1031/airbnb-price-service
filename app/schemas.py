from pydantic import BaseModel

class RentalRequest(BaseModel):
    accommodates: int
    bathrooms: float
    bedrooms: float
    beds: float
    minimum_nights: int
    availability_365: int
    number_of_reviews: int
    review_scores_rating: float
    reviews_per_month: float
    room_type: str
    property_type: str
    neighbourhood_cleansed: str
    cancellation_policy: str
    instant_bookable: str