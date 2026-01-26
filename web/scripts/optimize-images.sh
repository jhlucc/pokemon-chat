#!/bin/bash
# 图片优化脚本
# 使用前需要安装: npm install -g sharp-cli 或使用 ImageMagick

# 创建优化后的图片目录
mkdir -p public/optimized

echo "=== Pokemon-Chat 图片优化 ==="
echo ""

# 检查是否安装了必要的工具
if command -v convert &> /dev/null; then
    echo "✓ ImageMagick 已安装"
    USE_IMAGEMAGICK=true
else
    echo "✗ ImageMagick 未安装"
    USE_IMAGEMAGICK=false
fi

if command -v cwebp &> /dev/null; then
    echo "✓ WebP 工具已安装"
    USE_WEBP=true
else
    echo "✗ WebP 工具未安装 (可选)"
    USE_WEBP=false
fi

echo ""
echo "=== 开始优化 ==="

# 优化 home.jpg (7.8MB -> <200KB)
if [ -f "public/home.jpg" ] && [ "$USE_IMAGEMAGICK" = true ]; then
    echo "优化 home.jpg..."
    # 压缩并调整尺寸
    convert public/home.jpg -resize 1920x1080\> -quality 75 -strip public/optimized/home.jpg

    # 生成响应式版本
    convert public/home.jpg -resize 1200x -quality 75 -strip public/optimized/home-1200w.jpg
    convert public/home.jpg -resize 800x -quality 75 -strip public/optimized/home-800w.jpg
    convert public/home.jpg -resize 400x -quality 75 -strip public/optimized/home-400w.jpg

    # 生成 WebP 版本 (更小)
    if [ "$USE_WEBP" = true ]; then
        cwebp -q 80 public/optimized/home.jpg -o public/optimized/home.webp
        cwebp -q 80 public/optimized/home-1200w.jpg -o public/optimized/home-1200w.webp
        cwebp -q 80 public/optimized/home-800w.jpg -o public/optimized/home-800w.webp
        cwebp -q 80 public/optimized/home-400w.jpg -o public/optimized/home-400w.webp
    fi

    echo "✓ home.jpg 优化完成"
fi

# 优化 logo.png (460KB -> <50KB)
if [ -f "public/logo.png" ] && [ "$USE_IMAGEMAGICK" = true ]; then
    echo "优化 logo.png..."
    convert public/logo.png -resize 256x256\> -quality 85 -strip PNG8:public/optimized/logo.png

    if [ "$USE_WEBP" = true ]; then
        cwebp -q 85 public/logo.png -o public/optimized/logo.webp
    fi

    echo "✓ logo.png 优化完成"
fi

# 优化其他 PNG 图片
for img in public/*.png; do
    if [ -f "$img" ] && [ "$USE_IMAGEMAGICK" = true ]; then
        filename=$(basename "$img")
        if [ "$filename" != "logo.png" ]; then
            echo "优化 $filename..."
            convert "$img" -quality 85 -strip "public/optimized/$filename"
        fi
    fi
done

echo ""
echo "=== 优化完成 ==="
echo ""
echo "优化后的图片位于: public/optimized/"
echo ""
echo "替换步骤:"
echo "1. 检查 public/optimized/ 目录中的图片质量"
echo "2. 备份原始图片: mv public/home.jpg public/home.jpg.bak"
echo "3. 使用优化版本: cp public/optimized/home.jpg public/"
echo ""
echo "响应式图片使用示例:"
echo '<picture>'
echo '  <source srcset="/home.webp" type="image/webp">'
echo '  <source srcset="/home-400w.jpg 400w, /home-800w.jpg 800w, /home-1200w.jpg 1200w"'
echo '          sizes="(max-width: 600px) 400px, (max-width: 1200px) 800px, 1200px">'
echo '  <img src="/home.jpg" alt="Home" loading="lazy">'
echo '</picture>'
