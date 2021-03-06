const webpack = require('webpack');
const packageJson = require('./package.json');
const CleanWebpackPlugin = require('clean-webpack-plugin');

module.exports = {
  mode: 'production',
  devtool: 'source-map',
  entry: {
    app: [
      './src/index.tsx'
    ]
  },
  resolve: {
    extensions: [".ts", ".tsx", ".js", ".json"]
  },
  output: {
    path: __dirname + '/dist',
    filename: 'bundle.js',
    publicPath: __dirname + '/dist'
  },
  plugins: [
    new CleanWebpackPlugin(['dist'])
  ],
  module: {
    rules: [
      { 
        test: /\.tsx?$/, 
        loader: "awesome-typescript-loader",
        exclude: /node_modules/ 
      },
      { 
        test: /\.js?$/, 
        loader: "source-map-loader" 
      },
      {
        test: /\.(png|svg|jpg|gif)$/,
        use: ["file-loader"]
      },
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/,
        use: ["file-loader"]
      }
    ]
  },
  // When importing a module whose path matches one of the following, just
  // assume a corresponding global variable exists and use that instead.
  // This is important because it allows us to avoid bundling all of our
  // dependencies, which allows browsers to cache those libraries between builds.
  externals: {
    "react": "React",
    "react-dom": "ReactDOM"
  }
};
