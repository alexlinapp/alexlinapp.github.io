Personal Website




### Notes
For me.

- Change `_config.yml` defaults to page rules
- Exclude/Include:
    - Tells Jekyll should I read this file?
    - If excluded, does not even read the file
- Published controls:
    - After reading the file, determines whether or not the page should be published


Full Jekyll Sequence: 
```
Start Jekyll
    ↓
Load configuration from _config.yml
    ↓
Set up the site and plugins
    ↓
Read content files using those settings
    ↓
Run generators
    ↓
Render: Liquid → Markdown conversion → layouts
    ↓
Write the finished website
```
See Jekyll Rendering [docs](https://jekyllrb.com/docs/rendering-process/)









#### More Config

- Disabled Jekyll Archives for now
    - [Jekyll Archives Plugin](https://github.com/george-gca/jekyll-archives-v2#about-jekyll-archives)

- Jekyll Site Variables:
    - [docs](https://jekyllrb.com/docs/variables/#site-variables)