"""COMPASS abstract Service class"""

import asyncio
import logging
from abc import ABC, abstractmethod

from compass.services.queues import get_service_queue
from compass.exceptions import COMPASSNotInitializedError


logger = logging.getLogger(__name__)
MISSING_SERVICE_MESSAGE = """Must initialize the queue for {service_name!r}.
You can likely use the following code structure to fix this:

    from compass.services.provider import RunningAsyncServices

    services = [
        ...
        {service_name}(...),
        ...
    ]
    async with RunningAsyncServices(services):
        # function call here

"""


class Service(ABC):
    """Abstract base class for a Service that can be queued to run"""

    MAX_CONCURRENT_JOBS = 10_000
    """Max number of concurrent job submissions."""

    @classmethod
    def _queue(cls):
        """Get queue for class."""
        service_name = cls.__name__
        queue = get_service_queue(service_name)
        if queue is None:
            msg = MISSING_SERVICE_MESSAGE.format(service_name=service_name)
            raise COMPASSNotInitializedError(msg)
        return queue

    @classmethod
    async def call(cls, *args, **kwargs):
        """Call the service

        Parameters
        ----------
        *args, **kwargs
            Positional and keyword arguments to be passed to the
            underlying service processing function.

        Returns
        -------
        obj
            A response object from the underlying service.
        """
        fut = asyncio.Future()
        outer_task_name = asyncio.current_task().get_name()
        await cls._queue().put((fut, outer_task_name, args, kwargs))
        return await fut

    @property
    def name(self):
        """str: Service name used to pull the correct queue object"""
        return self.__class__.__name__

    async def process_using_futures(self, fut, *args, **kwargs):
        """Process a call to the service

        Parameters
        ----------
        fut : asyncio.Future
            A future object that should get the result of the processing
            operation. If the processing function returns ``answer``,
            this method should call ``fut.set_result(answer)``.
        **kwargs
            Keyword arguments to be passed to the
            underlying processing function.
        """

        try:
            response = await self.process(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
            return

        fut.set_result(response)

    def acquire_resources(self):  # noqa: B027
        """Use this method to allocate resources, if needed"""

    def release_resources(self):  # noqa: B027
        """Use this method to clean up resources, if needed"""

    @property
    @abstractmethod
    def can_process(self):
        """Check if process function can be called.

        This should be a fast-running method that returns a boolean
        indicating whether or not the service can accept more
        processing calls.
        """

    @abstractmethod
    async def process(self, *args, **kwargs):
        """Process a call to the service.

        Parameters
        ----------
        *args, **kwargs
            Positional and keyword arguments to be passed to the
            underlying processing function.
        """


class LLMService(Service):
    """Base class for LLm service

    This service differs from other services in that it must be used
    as an object, not as a class. that is, users must initialize it and
    pass it around in functions in order to use it.
    """

    def __init__(self, model_name, rate_limit, rate_tracker, service_tag=None):
        """

        Parameters
        ----------
        model_name : str
            Name of model being used.
        rate_limit : int or float
            Max usage per duration of the rate tracker. For example,
            if the rate tracker is set to compute the total over
            minute-long intervals, this value should be the max usage
            per minute.
        rate_tracker : `TimeBoundedUsageTracker`
            A TimeBoundedUsageTracker instance. This will be used to
            track usage per time interval and compare to `rate_limit`.
        service_tag : str, optional
            optional tag to use to distinguish service (i.e. make unique
            from other services). Must set this if multiple models with
            the same name are run concurrently. By default, ``None``.
        """
        self.model_name = model_name
        self.rate_limit = rate_limit
        self.rate_tracker = rate_tracker
        self.service_tag = service_tag or ""

    @property
    def can_process(self):
        """bool: Check if usage is under the rate limit"""
        return self.rate_tracker.total < self.rate_limit

    @property
    def name(self):
        """str: Unique service name used to pull the correct queue"""
        return f"{self.__class__.__name__}-{self.model_name}{self.service_tag}"

    def _queue(self):
        """Get queue for class"""
        queue = get_service_queue(self.name)
        if queue is None:
            msg = MISSING_SERVICE_MESSAGE.format(service_name=self.name)
            raise COMPASSNotInitializedError(msg)
        return queue

    async def call(self, *args, **kwargs):
        """Call the service

        Parameters
        ----------
        *args, **kwargs
            Positional and keyword arguments to be passed to the
            underlying service processing function.

        Returns
        -------
        obj
            A response object from the underlying service.
        """
        fut = asyncio.Future()
        outer_task_name = asyncio.current_task().get_name()
        await self._queue().put((fut, outer_task_name, args, kwargs))
        return await fut
