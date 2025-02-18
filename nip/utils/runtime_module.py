"""Implementation of the runtime modules from PyExt.

This is copied from the PyExt project, but we replace `inspect.getargspec` with
`inspect.getfullargspec` because the former is not available in Python 3.11.


Copyright (C) 2015 Ryan Gonzalez


Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

g_backup = globals().copy()

import sys, inspect, types, functools

def _targspec(func, specs, attr='__orig_arg__'):
    if hasattr(func, '__is_overload__') and func.__is_overload__:
        return getattr(func, attr)
    return specs(func)

def set_docstring(doc):
    '''A simple decorator to set docstrings.

       :param doc: The docstring to tie to the function.

       Example::

          @set_docstring('This is a docstring')
          def myfunc(x):
              pass'''
    def wrap(f):
        f.__doc__ = doc
        return f
    return wrap

_modify_function_doc = '''
Creates a copy of a function, changing its attributes.

:param globals: Will be added to the function's globals.

:param name: The new function name. Set to ``None`` to use the function's original name.

:param code: The new function code object. Set to ``None`` to use the function's original code object.

:param defaults: The new function defaults. Set to ``None`` to use the function's original defaults.

:param closure: The new function closure. Set to ``None`` to use the function's original closure.

.. warning:: This function can be potentially dangerous.
'''

def copyfunc(f):
   '''Copies a funcion.

      :param f: The function to copy.

      :return: The copied function.

      .. deprecated:: 0.4
         Use :func:`modify_function` instead.
      '''
   return modify_function(f)

if sys.version_info.major == 3:
    @set_docstring(_modify_function_doc)
    def modify_function(f, globals={}, name=None, code=None, defaults=None,
                        closure=None):
        if code is None: code = f.__code__
        if name is None: name = f.__name__
        if defaults is None: defaults = f.__defaults__
        if closure is None: closure = f.__closure__
        newf = types.FunctionType(code, dict(f.__globals__, **globals), name=name,
                                  argdefs=defaults, closure=closure)
        newf.__dict__.update(f.__dict__)
        return newf
    argspec = inspect.getfullargspec
    ofullargspec = inspect.getfullargspec
    def _fullargspec(func):
        return _targspec(func, ofullargspec)
    inspect.getfullargspec = _fullargspec
    def _exec(m,g): exec(m,g)
else:
    @set_docstring(_modify_function_doc)
    def modify_function(f, globals={}, name=None, code=None, defaults=None,
                        closure=None):
        if code is None: code = f.func_code
        if name is None: name = f.__name__
        if defaults is None: defaults = f.func_defaults
        if closure is None: closure = f.func_closure
        newf = types.FunctionType(code, dict(f.func_globals, **globals),
                                  name=name, argdefs=defaults, closure=closure)
        newf.__dict__.update(f.__dict__)
        return newf
    argspec = inspect.getfullargspec
    eval(compile('def _exec(m,g): exec m in g', '<exec>', 'exec'))

def _gettypes(args):
    return tuple(map(type, args))

oargspec = inspect.getfullargspec

def _argspec(func):
    return _targspec(func, oargspec)

inspect.getfullargspec = _argspec

try:
    import IPython
except ImportError:
    IPython = None
else:
    # Replace IPython's argspec
    oipyargspec = IPython.core.oinspect.getfullargspec
    def _ipyargspec(func):
        return _targspec(func, oipyargspec, '__orig_arg_ipy__')
    IPython.core.oinspect.getfullargspec = _ipyargspec

class overload(object):
    '''Simple function overloading in Python.'''
    @classmethod
    def argc(self, argc=None):
        '''Overloads a function based on the specified argument count.

           :param argc: The argument count. Defaults to ``None``. If ``None`` is given, automatically compute the argument count from the given function.

           .. note::

              Keyword argument counts are NOT checked! In addition, when the argument count is automatically calculated, the keyword argument count is also ignored!

           Example::

               @overload.argc()
               def func(a):
                   print 'Function 1 called'

               @overload.argc()
               def func(a, b):
                   print 'Function 2 called'

               func(1) # Calls first function
               func(1, 2) # Calls second function
               func() # Raises error
               '''
        # Python 2 UnboundLocalError fix
        argc = {'argc': argc}
        def wrap(f):
            if argc['argc'] is None:
                argc['argc'] = len(argspec(f).args)
            try:
                st = inspect.stack()[1][0]
                oldf = dict(st.f_globals, **st.f_locals)[f.__name__]
            except KeyError: pass
            else:
                if hasattr(oldf, '__pyext_overload_basic__'):
                    globls = oldf.__globals__ if sys.version_info.major == 3\
                             else oldf.func_globals
                    globls['overloads'][argc['argc']] = f
                    return oldf
            @functools.wraps(f)
            def newf(*args, **kwargs):
                if len(args) not in overloads:
                    raise TypeError(
                        "No overload of function '%s' that takes %d args" % (
                            f.__name__, len(args)))
                return overloads[len(args)](*args, **kwargs)
            overloads = {}
            overloads[argc['argc']] = f
            newf = modify_function(newf, globals={'overloads': overloads})
            newf.__pyext_overload_basic__ = None
            newf.__orig_arg__ = argspec(f)
            if IPython:
                newf.__orig_arg_ipy__ = IPython.core.oinspect.getfullargspec(f)
            return newf
        return wrap
    @classmethod
    def args(self, *argtypes, **kw):
        '''Overload a function based on the specified argument types.

           :param argtypes: The argument types. If None is given, get the argument types from the function annotations(Python 3 only)
           :param kw: Can only contain 1 argument, `is_cls`. If True, the function is assumed to be part of a class.

           Example::

               @overload.args(str)
               def func(s):
                   print 'Got string'

               @overload.args(int, str)
               def func(i, s):
                   print 'Got int and string'

               @overload.args()
               def func(i:int): # A function annotation example
                   print 'Got int'

               func('s')
               func(1)
               func(1, 's')
               func(True) # Raises error
            '''

        # XXX: some of this should be moved to a utility class
        # It's duplicated from overload.argc
        # Python 2 UnboundLocalError fix...again!
        argtypes = {'args': tuple(argtypes)}
        def wrap(f):
            if len(argtypes['args']) == 1 and argtypes['args'][0] is None:
                aspec = argspec(f)
                argtypes['args'] = tuple(map(lambda x: x[1], sorted(
                    aspec.annotations.items(),
                    key=lambda x: aspec.args.index(x[0]))))
            try:
                st = inspect.stack()[1][0]
                oldf = dict(st.f_globals, **st.f_locals)[f.__name__]
            except KeyError: pass
            else:
                if hasattr(oldf, '__pyext_overload_args__'):
                    globls = oldf.__globals__ if sys.version_info.major == 3\
                             else oldf.func_globals
                    globls['overloads'][argtypes['args']] = f
                    return oldf
            @functools.wraps(f)
            def newf(*args):
                if len(kw) == 0:
                    cargs = args
                elif len(kw) == 1 and 'is_cls' in kw and kw['is_cls']:
                    cargs = args[1:]
                else:
                    raise ValueError('Invalid keyword args specified')
                types = _gettypes(cargs)
                if types not in overloads:
                    raise TypeError(\
                        "No overload of function '%s' that takes: %s" % (
                            f.__name__, types))
                return overloads[types](*args)
            overloads = {}
            overloads[argtypes['args']] = f
            newf = modify_function(newf, globals={'overloads': overloads})
            newf.__pyext_overload_args__ = None
            newf.__orig_arg__ = argspec(f)
            if IPython:
                newf.__orig_arg_ipy__ = IPython.core.oinspect.getfullargspec(f)
            return newf
        return wrap

class _RuntimeModule(object):
    'Create a module object at runtime and insert it into sys.path. If called, same as :py:func:`from_objects`.'
    def __call__(self, *args, **kwargs):
        return self.from_objects(*args, **kwargs)
    @staticmethod
    @overload.argc(1)
    def from_objects(name, **d):
        return _RuntimeModule.from_objects(name, '', **d)
    @staticmethod
    @overload.argc(2)
    def from_objects(name, docstring, **d):
        '''Create a module at runtime from `d`.

           :param name: The module name.

           :param docstring: Optional. The module's docstring.

           :param \*\*d: All the keyword args, mapped from name->value.

           Example: ``RuntimeModule.from_objects('name', 'doc', a=1, b=2)``'''
        module = types.ModuleType(name, docstring)
        module.__dict__.update(d)
        module.__file__ = '<runtime_module>'
        sys.modules[name] = module
        return module
    @staticmethod
    @overload.argc(2)
    def from_string(name, s):
        return _RuntimeModule.from_string(name, '', s)
    @staticmethod
    @overload.argc(3)
    def from_string(name, docstring, s):
        '''Create a module at runtime from `s``.

           :param name: The module name.

           :param docstring: Optional. The module docstring.

           :param s: A string containing the module definition.'''
        g = {}
        _exec(s, g)
        return _RuntimeModule.from_objects(name, docstring,
                    **dict(filter(lambda x: x[0] not in g_backup, g.items())))

RuntimeModule = _RuntimeModule()