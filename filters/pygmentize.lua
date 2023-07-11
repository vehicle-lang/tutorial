if FORMAT:match 'html' then
    function Meta(el)
        local pygmentize = io.popen("pygmentize -S default -f html", "r")
        el["highlighting-css"] = pygmentize:read("a")
        return el
    end

    function CodeBlock(el)
        if el ~= nil and el.attr ~= nil and el.attr.classes ~= nil and pandoc.utils.type(el.attr.classes) == 'List' and
            el.attr.classes:includes('vehicle') then
            local specfile = os.tmpname()
            local specfile_output = io.output(specfile)
            specfile_output:write(el.text)
            specfile_output:close()
            local pygmentize = io.popen("pygmentize -l vehicle -f html -O nowrap=true " .. specfile, "r")
            local spechtml = '<pre class="vehicle"><code>' .. pygmentize:read("a") .. '</code></pre>'
            return pandoc.RawBlock('html', spechtml)
        end
    end
end
