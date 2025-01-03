#!/usr/bin/env python3
import typer
from typing_extensions import Annotated

app = typer.Typer()

@app.command()
def main(
    path: Annotated[str, typer.Argument(help="Input FITS file.")], 
    index: Annotated[int, typer.Argument(help="The HDU index (0-indexed) to read from.")] = None
):
    """Define a CasJobs-compatible SQL table schema from an Astra FITS file."""

    import re
    from astropy.io import fits

    camel_case_to_snake_case = lambda text: re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()

    def get_meta(path, index):
        basename = path.split("/")[-1]
        parts = camel_case_to_snake_case(basename.split("-")[0]).split("_")
        star_or_visit = [sv for sv in ("star", "visit") if sv in parts]
        if len(star_or_visit) > 1:
            raise ValueError(f"Cannot determine if {basename} is a star or visit file.")
        elif len(star_or_visit) == 0:
            typer.echo(f"Cannot determine if {path}:{index} is a star or visit file. Assuming star.")
            star_or_visit = ["star"]
            pipeline_name = "_".join(parts[1:]) # astra best
            instrument_name = "all"
        else:
            spectrum_type, = star_or_visit
            pipeline_name = "_".join(parts[parts.index(spectrum_type) + 1:])        
            instrument_name = {1: "boss", 2: "apogee"}.get(index, "unknown")

        return dict(
            spectrum_type=star_or_visit[0],
            pipeline_name=pipeline_name,
            instrument_name=instrument_name
        )

    FORMAT_MAP = {'L': 'bit', 'K': 'bigint', 'J': 'int', 'I': 'smallint', 'A': 'varchar',
              'D': 'real', 'E': 'float', 'P': 'array', 'Q': 'array', 'B': 'bit'}

    sql_create, any_data_in_any_hdu = ("", False)
    with fits.open(path) as image:
        indices = (index, ) if index is not None else range(len(image))

        for i in indices:
            hdu = image[i]

            if hdu.data is None or len(hdu.data) == 0:
                if index is None:
                    continue
                else:
                    raise ValueError(f"HDU {index} of {path} has no data.")
            else:
                any_data_in_any_hdu = True
            
            table_name = "{pipeline_name}_{instrument_name}_{spectrum_type}".format(**get_meta(path, i))

            sql_create += (
                f"// Created from HDU {i} in {path}\n"
                f"CREATE TABLE {table_name} (\n"
            )

            cards = iter(hdu.header.cards)

            for col in hdu.data.columns:
                field_type = FORMAT_MAP[col.format[-1]]
                if field_type == "varchar":
                    field_length = col.format[:-1]
                    field_type = f"{field_type}({field_length})"

                for key, value, comment in cards:
                    if value == col.name:
                        break
                else:
                    raise ValueError(f"Cannot find header for column {col.name}")

                # break out the units from the comment
                if "[" not in comment:
                    unit = None
                    desc = f"--/D {comment}"
                else:
                    # Don't recognize [*/H] or [*/Fe] as units
                    unit = comment.split("[")[-1]
                    unit = unit.split("]")[0]
                    
                    if unit.endswith("/H") or unit.endswith("/Fe"):
                        unit = "dex"
                        parsed_comment = comment
                    else:            
                        parsed_comment = comment[:-(2 + len(unit))]
                    desc = f"--/U {unit} --/D {parsed_comment}"
                sql_create += f"  {col.name} {field_type} NOT NULL, {desc}\n"

            sql_create += ");"
    
    if not any_data_in_any_hdu:
        typer.error(f"No data found in any HDU of {path}")

    typer.echo(sql_create)
    

if __name__ == "__main__":
    app.run()