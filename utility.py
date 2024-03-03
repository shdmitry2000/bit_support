import asyncio
import inspect
import traceback
import multiprocessing
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    ParamSpec,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
    reveal_type,
)

from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")


def is_coroutine(
    func: Union[Callable[P, R], Callable[P, Awaitable[R]]]
) -> TypeGuard[Callable[P, Awaitable[R]]]:
    return inspect.iscoroutinefunction(func)

def is_running_in_asyncio() -> bool:
    if hasattr(asyncio, 'get_running_loop'):
        try:
            loop = asyncio.get_running_loop()
            return loop is not None
        except RuntimeError:
            return False
    else:
        return False
    
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper 

@overload
def asyncdecorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    ...


@overload
def asyncdecorator(func: Callable[P, R]) -> Callable[P, R]:
    ...


def asyncdecorator(
    func: Union[Callable[P, R], Callable[P, Awaitable[R]]]
) -> Union[Callable[P, R], Callable[P, Awaitable[R]]]:
    if is_coroutine(func):

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return async_wrapper

    else:
        func = cast(Callable[P, R], func)

        async def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if is_running_in_asyncio():
                loop = asyncio.get_running_loop()
                return  loop.run_in_executor(None, func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
   

        return sync_wrapper
    
def syncdecorator(
    func: Union[Callable[P, R], Callable[P, Awaitable[R]]]
) -> Union[Callable[P, R], Callable[P, Awaitable[R]]]:
    if is_coroutine(func):

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await func(*args, **kwargs)

        return async_wrapper
    else:
        func = cast(Callable[P, R], func)

        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return sync_wrapper
    
    
def sync_to_async(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_running_in_asyncio():
            loop = asyncio.get_running_loop()
            return  loop.run_in_executor(None, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
            
    return wrapper







# def run_async_with_callback(callback):
#     def inner(func):
#         def wrapper(*args, **kwargs):
#             def __exec():
#                 out = func(*args, **kwargs)
#                 callback(out)
#                 return out

#             return asyncio.get_event_loop().run_in_executor(None, __exec)

#         return wrapper

#     return inner


# def run_method(class_instance, method_name, *args, **kwargs):
#         method = getattr(class_instance, method_name, None)
#         if method is not None:
#             return method(*args, **kwargs)
#         else:
#             print(f"No method named {method_name} found in the class")



# @syncdecorator
# def sync_fn_any(x):
#     return x


# @syncdecorator
# async def async_fn_any(x):
#     return x


# sync_fn_any_result =  sync_fn_any(1)
# async_fn_any_result =  async_fn_any(1)
# print( sync_fn_any_result)
# print( sync_fn_any_result)
# reveal_type(sync_fn_any_result)    # Awaitable[List[Unknown]]    -- Wanted List[Any] or List[Unknown]
# reveal_type(async_fn_any_result)  # Awaitable[List[Unknown]]