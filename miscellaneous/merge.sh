
for dir in */; do
    dirname=${dir%/}
    newname=${dirname// /_}
    
    if [ -d "$newname" ] && [ "$dirname" != "$newname" ]; then
        # If directory with underscore name exists, merge contents
        echo "Merging: $dirname into $newname"
        cp -r "$dirname"/* "$newname"/
        rm -r "$dirname"
    elif [ "$dirname" != "$newname" ]; then
        # If no existing directory, just rename
        mv "$dirname" "$newname"
        echo "Renamed: $dirname -> $newname"
    fi
done