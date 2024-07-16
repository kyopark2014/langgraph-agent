const aws = require('aws-sdk');

const s3 = new aws.S3();

const bucketName = process.env.bucketName;
const s3_prefix = process.env.s3_prefix;

const URL_EXPIRATION_SECONDS = 300;

exports.handler = async (event, context) => {
    //console.log('## ENVIRONMENT VARIABLES: ' + JSON.stringify(process.env));
    //console.log('## EVENT: ' + JSON.stringify(event));
        
    let filename = event['filename'];
    let contentType = event['contentType'];

    const s3Params = {
        Bucket: bucketName,
        Key: s3_prefix+'/'+filename,
        Expires: URL_EXPIRATION_SECONDS,
        ContentType: contentType,
    };
    console.log('s3Params: ', JSON.stringify(s3Params));

    const uploadURL = await s3.getSignedUrlPromise('putObject', s3Params);
    console.log('uploadURL: ', uploadURL);

    const response = {
        statusCode: 200,
        body: JSON.stringify({
            Bucket: bucketName,
            Key: s3_prefix+'/'+filename,
            Expires: URL_EXPIRATION_SECONDS,
            ContentType: contentType,
            UploadURL: uploadURL
        })
    };
    return response;
};