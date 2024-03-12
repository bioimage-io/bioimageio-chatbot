"""Jsonschema to pydantic schema from https://github.com/c32168/dyntamic"""
from typing import Annotated, Union, Any, Optional

import typing
from pydantic import create_model
from pydantic.fields import Field, PydanticUndefined

Model = typing.TypeVar('Model', bound='BaseModel')


class DyntamicFactory:

    TYPES = {
        'string': str,
        'array': list,
        'boolean': bool,
        'integer': int,
        'float': float,
        'number': float,
        'null': None,
    }

    def __init__(self,
                 json_schema: dict,
                 base_model: Union[type[Model], tuple[type[Model], ...], None] = None,
                 ref_template: str = "definitions"
                 ) -> None:
        """
        Creates a dynamic pydantic model from a JSONSchema, dumped from and existing Pydantic model elsewhere.
            JSONSchema dump must be called with ref_template='{model}' like:

            SomeSampleModel.model_json_schema(ref_template='{model}')
            Use:
            >> _factory = DyntamicFactory(schema)
            >> _factory.make()
            >> _model = create_model(_factory.class_name, **_factory.model_fields)
            >> _instance = dynamic_model.model_validate(json_with_data)
            >> validated_data = model_instance.model_dump()
        """
        self.class_name = json_schema.get('title')
        self.description = json_schema.get('description')
        self.class_type = json_schema.get('type')
        self.required = json_schema.get('required', [])
        self.default = json_schema.get('default')
        self.raw_fields = json_schema.get('properties')
        self.ref_template = ref_template
        self.definitions = json_schema.get(ref_template)
        self.fields = {}
        self.model_fields = {}
        self._base_model = base_model
        
    def get_factory(self, field_name, field) -> Any:
        """Get the factory for a given type"""
        f_type = field.get('type')
        if f_type is None and 'anyOf' in field:
            factory = tuple([self.get_factory(None, t) for t in field.get('anyOf')])
            if None in factory and len(factory) == 2:
                if field_name and field_name not in self.required:
                    factory = [f for f in factory if f is not None][0]
                else:
                    factory = Optional[[f for f in factory if f is not None][0]]
            else:
                factory = Union[factory]
        else:
            factory = self.TYPES.get(f_type)
        return factory

    def make(self) -> Model:
        """Factory method, dynamically creates a pydantic model from JSON Schema"""
        for field in self.raw_fields:
            if field not in self.required:
                default = self.raw_fields[field].get('default')
            else:
                default = PydanticUndefined
            if '$ref' in self.raw_fields[field]:
                model_name = self.raw_fields[field].get('$ref')
                # resolve $ref
                # consider all the cases in standard json schema
                
                if model_name.startswith('#/'):
                    model_name = model_name.replace('#/', '')
                elif model_name.startswith('#'):
                    model_name = model_name.replace('#', '')
                
                if model_name.startswith(self.ref_template+"/"):
                    model_name = model_name.replace(self.ref_template+"/", '')

                self._make_nested(model_name, field, default)
            else:
                factory = self.get_factory(field, self.raw_fields[field])
                if factory is None:
                    factory = Any
                if factory == list:
                    items = self.raw_fields[field].get('items')
                    if self.ref_template in items:
                        self._make_nested(items.get(self.ref_template), field)
                
                self._make_field(factory, field, self.raw_fields.get('title'), self.raw_fields.get(field).get('description'), default=default)
        model = create_model(self.class_name, __base__=self._base_model, **self.model_fields)
        model.__doc__ = self.description
        return model

    def _make_nested(self, model_name: str, field, default) -> None:
        """Create a nested model"""
        level = DyntamicFactory({self.ref_template: self.definitions} | self.definitions.get(model_name),
                                ref_template=self.ref_template)
        level.make()
        model = create_model(model_name, **level.model_fields)
        model.__doc__ = level.description
        self._make_field(model, field, field, level.description, default)

    def _make_field(self, factory, field, alias, description, default) -> None:
        """Create an annotated field"""
        # if field not in self.required:
        #     factory_annotation = Annotated[Union[factory, None], factory]
        # else:
        factory_annotation = factory
        self.model_fields[field] = (
            Annotated[factory_annotation, Field(default_factory=None, alias=alias, description=description)], default)

def json_schema_to_pydantic_model(schema):
    f = DyntamicFactory(schema)
    return f.make()

if __name__ == "__main__":
    input_schema = {
        "title": "RunMacro",
        "description": "Run a macro",
        "type": "object",
        "properties": {
            "macro": {
                "type": "string",
                "description": "The macro to run"
            },
            "args": {"$ref": "#/definitions/Args"},
            "query": {"description": "The search query string.", "title": "Query", "type": "string"},
            "pageSize": {"default": 10, "description": "Number of search results per page.", "exclusiveMinimum": 0, "title": "Pagesize", "type": "integer"},
            "page": {"default": 1, "description": "Page number of the search results.", "title": "Page", "type": "integer"},
            "sortOrder": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "descending", "description": "Sort order: ascending or descending.", "title": "Sortorder"}
        },
        "required": ["macro", "query"],
        "definitions": {
            "Args": {
                "title": "Args",
                "type": "object",
                "description": "Arguments for the macro",
                "properties": {
                    "arg1": {
                        "type": "string",
                        "description": "arg1"
                    }
                }
            }
        }
    }
    RunMacroClass = json_schema_to_pydantic_model(input_schema)
    assert RunMacroClass.__name__ == input_schema["title"]
    # assert RunMacroClass.__doc__ == input_schema["description"]
    m = RunMacroClass(macro="test", args={"test": "test"}, query="test")
    schema = RunMacroClass.model_json_schema()
    print(schema)
    assert schema['title'] == input_schema['title']
    assert schema['description'] == input_schema['description']
    assert schema['properties']['macro']["description"] == input_schema['properties']['macro']["description"]
    assert schema['properties']['args']['allOf'][0]['$ref'] == "#/$defs/Args"
    assert m.macro == "test"
    