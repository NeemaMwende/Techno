module.exports = {
  style: {
    postcss: {
      plugins: (plugins) => [
        require("tailwindcss")(),
        require("autoprefixer")(),
        ...plugins,
      ],
    },
  },
};
