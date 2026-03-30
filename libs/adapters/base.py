from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    def prepare(self, request_envelope):
        pass

    @abstractmethod
    async def invoke(self, prepared_request):
        pass

    @abstractmethod
    async def stream(self, prepared_request):
        pass

    @abstractmethod
    def normalize_response(self, raw_response):
        pass

    @abstractmethod
    def normalize_error(self, raw_error):
        pass
