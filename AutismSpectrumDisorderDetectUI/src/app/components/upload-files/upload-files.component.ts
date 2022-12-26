import { Component, OnInit } from '@angular/core';
import { UploadFilesService } from 'src/app/services/upload-files.service';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-upload-files',
  templateUrl: './upload-files.component.html',
  styleUrls: ['./upload-files.component.css']
})
export class UploadFilesComponent implements OnInit {
  selectedFiles: FileList;
  progressInfos = [];
  message = '';
  fileName = [];
  uploadStarted = false;
  uploadCompleted = "0";

  fileInfos: Observable<any>;

  constructor(private uploadService: UploadFilesService) { }

  ngOnInit(): void {
    this.fileInfos = this.uploadService.getFiles();
    this.uploadCompleted = "0";
    localStorage.setItem("completed", this.uploadCompleted);
  }

  selectFiles(event) {
    this.progressInfos = [];
    this.uploadCompleted = "0";
    localStorage.removeItem("completed");
    localStorage.setItem("completed", this.uploadCompleted);
    this.fileName = [];
    this.selectedFiles = event.target.files;
    for (let i = 0; i < this.selectedFiles.length; i++) {
      this.fileName.push(this.selectedFiles[i]['name']);
    }
  }

  upload(idx, file) {
    this.progressInfos[idx] = { value: 0, fileName: file.name };

    this.uploadService.upload(file).subscribe(
      event => {
        if (event.type === HttpEventType.UploadProgress) {
          this.progressInfos[idx].value = Math.round(100 * event.loaded / event.total);
        } else if (event instanceof HttpResponse) {
          this.fileInfos = this.uploadService.getFiles();
          this.fileInfos.subscribe((data) => {
            localStorage.setItem("files", this.fileName.toString());
          });
        }
      },
      err => {
        this.progressInfos[idx].value = 0;
        this.message = 'Could not upload the file:' + file.name;
      });
  }

  uploadFiles() {
    this.uploadStarted = true;
    this.message = '';
    for (let i = 0; i < this.selectedFiles.length; i++) {
      this.upload(i, this.selectedFiles[i]);
    }
    this.uploadCompleted = "1";
    localStorage.removeItem("completed");
    localStorage.setItem("completed", this.uploadCompleted);
  }
}
