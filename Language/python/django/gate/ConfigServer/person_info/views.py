from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView
from person_info.models import Person

class PersonList(ListView):
    model = Person
    template_name = ''