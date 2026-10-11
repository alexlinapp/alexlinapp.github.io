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





### Docker
- Build, recreate, start and attach containers:
    - `docker compose up`:
        - Also starts the container and etc
- Start the container for local:
    - `docker compose start jekyll`
    - `docker compose stop jekyll`
- Restart the `jekyll` container:
    - `docker compose restart jekyll`
- Check Status
    - `docker compose status`
    - `docker ps`



#### More Config

- Disabled Jekyll Archives for now
  - [Jekyll Archives Plugin](https://github.com/george-gca/jekyll-archives-v2#about-jekyll-archives)

- Jekyll Site Variables:
    - [docs](https://jekyllrb.com/docs/variables/#site-variables)





### References
- CSS, HTML Pain :(
    - https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/object-fit#values
