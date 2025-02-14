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
        indicating wether or not the service can accept more
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


class RateLimitedService(Service):
    """ABC representing a rate-limited service (e.g. OpenAI)"""

    def __init__(self, rate_limit, rate_tracker):
        """

        Parameters
        ----------
        rate_limit : int | float
            Max usage per duration of the rate tracker. For example,
            if the rate tracker is set to compute the total over
            minute-long intervals, this value should be the max usage
            per minute.
        rate_tracker : `TimeBoundedUsageTracker`
            A TimeBoundedUsageTracker instance. This will be used to
            track usage per time interval and compare to `rate_limit`.
        """
        self.rate_limit = rate_limit
        self.rate_tracker = rate_tracker

    @property
    def can_process(self):
        """Check if usage is under the rate limit."""
        return self.rate_tracker.total < self.rate_limit
