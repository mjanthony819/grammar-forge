"""Pydantic schemas used by the benchmark suite."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str
    age: int
    hobbies: List[str]


class Product(BaseModel):
    title: str
    price: float
    in_stock: bool
    tags: List[str]


class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    country: str


class Event(BaseModel):
    title: str
    date: str
    location: str
    attendees: int


class Recipe(BaseModel):
    name: str
    prep_minutes: int
    ingredients: List[str]
    vegetarian: bool


class Book(BaseModel):
    title: str
    author: str
    year: int
    genres: List[str]


class Review(BaseModel):
    movie: str
    rating: float = Field(ge=0, le=10)
    recommend: bool
    comment: str


class OrderLine(BaseModel):
    sku: str
    quantity: int
    unit_price: float


class Order(BaseModel):
    order_id: str
    lines: List[OrderLine]
    total: float


class UserProfile(BaseModel):
    username: str
    email: str
    age: Optional[int]
    premium: bool


class GeoPoint(BaseModel):
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Playlist(BaseModel):
    name: str
    tracks: List[str]
    public: bool


class Invoice(BaseModel):
    invoice_no: str
    amount: float
    currency: str
    paid: bool


TASKS = [
    ("person", Person, "a person: name (string), age (integer), hobbies (array of strings)"),
    ("product", Product, "a product: title (string), price (number), in_stock (boolean), tags (array of strings)"),
    ("address", Address, "an address: street, city, zip_code, country (all strings)"),
    ("event", Event, "an event: title (string), date (string), location (string), attendees (integer)"),
    ("recipe", Recipe, "a recipe: name (string), prep_minutes (integer), ingredients (array of strings), vegetarian (boolean)"),
    ("book", Book, "a book: title (string), author (string), year (integer), genres (array of strings)"),
    ("review", Review, "a movie review: movie (string), rating (number 0-10), recommend (boolean), comment (string)"),
    ("order", Order, "an order: order_id (string), lines (array of objects with sku (string), quantity (integer), unit_price (number)), total (number)"),
    ("profile", UserProfile, "a user profile: username (string), email (string), age (integer or null), premium (boolean)"),
    ("geo", GeoPoint, "a geo point: name (string), latitude (number), longitude (number)"),
    ("playlist", Playlist, "a playlist: name (string), tracks (array of strings), public (boolean)"),
    ("invoice", Invoice, "an invoice: invoice_no (string), amount (number), currency (string), paid (boolean)"),
]
