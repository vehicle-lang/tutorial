if FORMAT:match 'latex' then
    function CodeBlock(el)
        if el ~= nil and el.attr ~= nil and el.attr.classes ~= nil and pandoc.utils.type(el.attr.classes) == 'List' and
            el.attr.classes:includes('vehicle') then
            return pandoc.RawBlock('latex', '\\begin{minted}{vehicle}\n' .. el.text .. '\n\\end{minted}')
        end
    end
end
